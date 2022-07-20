import os
import subprocess 

from i6_utils.bliss_corpus import Corpus

def bliss_to_tmp_data_dir(bliss_corpus_file):
    TMPDIR = "tmpdir"
    if not os.path.exists(TMPDIR):
        os.mkdir(TMPDIR)
    c = Corpus()
    c.load(bliss_corpus_file)
    wav_recordings = []
    print("convert to 16 kHz wav")
    for recording in c.all_recordings():
        name, ext = os.path.splitext(os.path.basename(recording.audio))
        outpath = os.path.join(TMPDIR, name + ".wav")
        cmd = ["sox", recording.audio, "-r", "16000", "-V1", outpath]
        subprocess.check_call(cmd)
        wav_recordings.append(outpath)

    print("run normalization")
    subprocess.check_call(["sv56_norm/sv56scripts/batch_normRMSE.sh", TMPDIR, os.getcwd()])

    print("create DATA folder")
    wav_folder = os.path.join(TMPDIR, "DATA", "wav")
    set_folder = os.path.join(TMPDIR, "DATA", "sets")
    set_file = os.path.join(set_folder, "val_mos_list.txt")
    subprocess.check_call(["mkdir", "-p", wav_folder])
    subprocess.check_call(["mkdir", "-p", set_folder])
    with open(set_file, "wt") as set_file:
        for recording in wav_recordings:
            norm_recording = os.path.basename(recording.replace(".wav", "_norm.wav"))
            set_file.write(f"{norm_recording},1.0\n")
            subprocess.check_call(["mv", os.path.join(TMPDIR, norm_recording), os.path.join(wav_folder, norm_recording)])
    return os.path.join(TMPDIR, "DATA")

