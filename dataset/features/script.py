import h5py
from shared.text_processing import TextTokenizer, PhonemeTokenizer

char_decoder    = TextTokenizer()
phoneme_encoder = PhonemeTokenizer()   # vocab_size ≈ 47, CTC-ready

if __name__ == "__main__":
    h5f = h5py.File('/home/ai/Projects/Mahsa/datasets/lrs2/lrs2_train_features_chunk1.h5', 'r')
    video_idx = 150

    keys = list(h5f.keys())
    print(h5f[keys[0]].keys())
    spec = h5f[f'{keys[video_idx]}/spec']
    print("spec shape:", spec.shape)
    print("path:", h5f.attrs.get(f"{keys[1]}/video_path", None))

    av_hubert = h5f[f'{keys[video_idx]}/visual_features'][:]

    print("av_hubert shape:", av_hubert.shape)
    print("av_hubert features min and max values:", av_hubert.min(), av_hubert.max())
    # ── character-level decode (fixed) ───────────────────────────────────────
    text_indices = h5f[f'{keys[video_idx]}/text'][:]
    char_text = ''.join(char_decoder.idx2char[int(c)] for c in text_indices
                        if int(c) != char_decoder.char2idx['<blank>'])
    print("char text :", char_text)

    # ── phoneme-level encode / decode ────────────────────────────────────────
    phone_indices = phoneme_encoder.encode(char_text, max_length=128)

    print("phoneme indices:", phone_indices)
    print("phonemes        :", phoneme_encoder.decode(phone_indices[0]))
    print("vocab size      :", phoneme_encoder.vocab_size)   # use as output_size in your CTC head
    phones = h5f[f'{keys[video_idx]}/phone_indices']
    print(phones)
    phone_str = ''.join([phoneme_encoder.idx2phone[p] for p in phones
                         if p != 0])
    print("phone str :", phone_str)
    phone_decoded = phoneme_encoder.decode(phones)
    print("phone decoded:", phone_decoded)