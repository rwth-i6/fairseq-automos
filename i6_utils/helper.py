import os
import subprocess
import sys

from i6_utils.bliss_corpus import Corpus


def create_data_dir(recordings, tmpdir="tmpdir"):
    if not os.path.exists(tmpdir):
        os.mkdir(tmpdir)
    print("convert to wav", file=sys.stderr)
    wav_recordings = []
    for recording in recordings:
        name, ext = os.path.splitext(os.path.basename(recording))
        outpath = os.path.join(tmpdir, name + ".wav")
        cmd = ["sox", recording, "-r", "16000", "-V1", outpath]
        subprocess.check_call(cmd)
        wav_recordings.append(outpath)

    print("run normalization", file=sys.stderr)
    root_dir = os.path.join(os.path.dirname(__file__), "..")
    norm_exec = os.path.join(root_dir, "sv56_norm/sv56scripts/batch_normRMSE.sh")
    subprocess.check_call([norm_exec, tmpdir, root_dir])

    print("create DATA folder", file=sys.stderr)
    wav_folder = os.path.join(tmpdir, "DATA", "wav")
    set_folder = os.path.join(tmpdir, "DATA", "sets")
    set_file = os.path.join(set_folder, "val_mos_list.txt")
    subprocess.check_call(["mkdir", "-p", wav_folder])
    subprocess.check_call(["mkdir", "-p", set_folder])
    with open(set_file, "wt") as set_file:
        for recording in wav_recordings:
            norm_recording = os.path.basename(recording.replace(".wav", "_norm.wav"))
            set_file.write(f"{norm_recording},1.0\n")
            subprocess.check_call(["mv", os.path.join(tmpdir, norm_recording), os.path.join(wav_folder, norm_recording)])
    return os.path.join(tmpdir, "DATA")


def bliss_to_tmp_data_dir(bliss_corpus_file, tmpdir="tmpdir"):
    c = Corpus()
    c.load(bliss_corpus_file)
    recordings = []
    for recording in c.all_recordings():
        assert os.path.exists(recording.audio)
        recordings.append(recording.audio)
    return create_data_dir(recordings, tmpdir=tmpdir)


def text_file_to_tmp_data_dir(text_file, tmpdir="tmpdir"):
    recordings = []
    with open(text_file, "rt") as f:
        for recording in f.readlines():
            recording = recording.strip()
            assert os.path.exists(recording)
            recordings.append(recording)
    return create_data_dir(recordings, tmpdir=tmpdir)




