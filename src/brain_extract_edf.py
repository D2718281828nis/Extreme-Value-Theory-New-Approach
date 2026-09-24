"""Пример 3 (мозг, стерео-ЭЭГ): извлечение компактного набора производных данных из открытой записи.

Исходные данные: запись стерео-ЭЭГ `sEEG-HFOs-8.edf` (Zenodo, DOI 10.5281/zenodo.21967993, CC BY 4.0),
103 сигнала по 256 Гц, 14 095 записей по 1 с, EDF+C с каналом аннотаций врача.

Файл EDF (743 МБ) слишком велик для переноса целиком, поэтому на компьютере автора этим скриптом
(только numpy, без MNE) строится производный набор, по которому считается весь пример:
  * logspec  [T, C, 67] float16 — log10 мощности 1-секундных окон (окно Ханна) в бинах 1..67 Гц
                                    (аппаратный ФНЧ записи — 67 Гц);
  * linelen, std_detr, ac1 [T, C]  — длина линии, ст. откл. после удаления линейного тренда, автокорреляция лаг 1;
  * raw_event [C, N] int16          — исходный сигнал 10000–10550 с (для каскада вовлечения и признаков узлов);
  * corr_base [C, C]                — корреляция Пирсона сигналов контактов на фоне 0–3600 с;
  * annotations                     — аннотации врача из канала EDF Annotations (cp1251).
T = 10 550 окон: запись обрезается на 10 550 с — после этой отметки в файле хирургическая каутеризация (§ 3.11).

Запуск: python3 brain_extract_edf.py <путь к sEEG-HFOs-8.edf> <выходной .npz>
"""
import hashlib, re, sys, time
import numpy as np

CROP = 10550
EVENT_RAW = (10000, 10550)
BASE = (0, 3600)
CONTACT = re.compile(r"^(?:EEG\s+)?([A-Za-z]+'?)\s*(\d+)$")


def header(path):
    with open(path, "rb") as f:
        h = f.read(256)
        hb = int(h[184:192]); nrec = int(h[236:244]); dur = float(h[244:252]); ns = int(h[252:256])
        f.seek(0); H = f.read(hb)
    def fld(off, w):
        return [H[256 + off + i * w: 256 + off + (i + 1) * w].decode("latin-1").strip() for i in range(ns)]
    o = 0; lab = fld(o, 16); o += 16 * ns; o += 80 * ns; dim = fld(o, 8); o += 8 * ns
    pmin = np.array(fld(o, 8), float); o += 8 * ns; pmax = np.array(fld(o, 8), float); o += 8 * ns
    dmin = np.array(fld(o, 8), float); o += 8 * ns; dmax = np.array(fld(o, 8), float); o += 8 * ns
    o += 80 * ns; spr = np.array(fld(o, 8), int)
    gain = (pmax - pmin) / (dmax - dmin); offs = pmin - gain * dmin
    return dict(hb=hb, nrec=nrec, dur=dur, ns=ns, lab=lab, dim=dim, spr=spr, gain=gain, offs=offs)


def main(path, out):
    t0 = time.time()
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            md5.update(chunk)
    H = header(path); spr = H["spr"]; assert len(set(spr)) == 1
    fs = int(spr[0] / H["dur"]); rec = int(spr.sum())
    mm = np.memmap(path, dtype="<i2", mode="r", offset=H["hb"], shape=(H["nrec"], rec))
    idx = [i for i, l in enumerate(H["lab"]) if CONTACT.match(l)]
    names = [H["lab"][i] for i in idx]
    ann_i = H["lab"].index("EDF Annotations")
    # --- аннотации (TAL): "+onset\x15dur\x14text\x14\x00"
    ann = []
    for r in range(H["nrec"]):
        b = mm[r, ann_i * fs:(ann_i + 1) * fs].tobytes()
        for tal in b.split(b"\x00"):
            if not tal.strip(b"\x00"):
                continue
            parts = tal.split(b"\x14")
            onset = parts[0].split(b"\x15")[0]
            for txt in parts[1:]:
                if txt:
                    ann.append((float(onset.decode("latin-1")), txt.decode("cp1251", "replace")))
    g = H["gain"][idx][:, None]; o = H["offs"][idx][:, None]
    win = np.hanning(fs); freqs = np.fft.rfftfreq(fs, 1 / fs); band = (freqs >= 1) & (freqs <= 67)
    T = CROP; C = len(idx)
    logspec = np.zeros((T, C, band.sum()), np.float16)
    linelen = np.zeros((T, C), np.float32); stdd = np.zeros((T, C), np.float32); ac1 = np.zeros((T, C), np.float32)
    S1 = np.zeros(C); S2 = np.zeros((C, C)); n = 0
    tt = np.arange(fs) - (fs - 1) / 2
    for r in range(T):
        x = (mm[r].reshape(-1, fs)[idx].astype(np.float64)) * g + o          # мкВ
        xc = x - x.mean(1, keepdims=True)
        beta = (xc * tt).sum(1, keepdims=True) / (tt ** 2).sum(); res = xc - beta * tt
        P = np.abs(np.fft.rfft(xc * win, axis=1)) ** 2
        logspec[r] = np.log10(P[:, band] + 1e-12).astype(np.float16)
        linelen[r] = np.abs(np.diff(x, axis=1)).mean(1); stdd[r] = res.std(1)
        ac1[r] = (res[:, 1:] * res[:, :-1]).sum(1) / ((res ** 2).sum(1) + 1e-12)
        if BASE[0] <= r < BASE[1]:
            S1 += x.sum(1); S2 += x @ x.T; n += fs
    mu = S1 / n; cov = S2 / n - np.outer(mu, mu); sd = np.sqrt(np.diag(cov)); corr = cov / np.outer(sd, sd)
    raw_event = np.concatenate([mm[r].reshape(-1, fs)[idx] for r in range(*EVENT_RAW)], axis=1).astype(np.int16)
    np.savez(out, names=np.array(names), fs=fs, crop=CROP, event_raw_start=EVENT_RAW[0], base=np.array(BASE),
             freqs=freqs[band], logspec=logspec, linelen=linelen, std_detr=stdd, ac1=ac1,
             raw_event=raw_event, gain=H["gain"][idx], offs=H["offs"][idx], corr_base=corr.astype(np.float32),
             ann_onset=np.array([a[0] for a in ann]), ann_text=np.array([a[1] for a in ann]),
             edf_md5=md5.hexdigest())
    print("md5", md5.hexdigest(), "contacts", C, "fs", fs, "annotations", len(ann), f"{time.time() - t0:.0f} s")
    for a in ann[:20]:
        print(a)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
