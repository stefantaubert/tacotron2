# Inference

## IPA
export text="examples/ipa/north_sven_orig.txt"
export text_name="ipa-north_sven_orig"
python -m src.cli.runner prepare-text-add --filepath=$text --prep_name=$prep_name --text_name=$text_name --lang=IPA
python -m src.cli.runner prepare-text-normalize --prep_name=$prep_name --text_name=$text_name
python -m src.cli.runner prepare-text-to-ipa --prep_name=$prep_name --text_name=$text_name

export text="examples/ipa/north_sven_v2.txt"
export text_name="ipa-north_sven_v2"
python -m src.cli.runner prepare-text-add --filepath=$text --prep_name=$prep_name --text_name=$text_name --lang=IPA
python -m src.cli.runner prepare-text-normalize --prep_name=$prep_name --text_name=$text_name
python -m src.cli.runner prepare-text-to-ipa --prep_name=$prep_name --text_name=$text_name

export text="examples/ipa/north_ger.txt"
export text_name="ipa-north_ger"
python -m src.cli.runner prepare-text-add --filepath=$text --prep_name=$prep_name --text_name=$text_name --lang=IPA
python -m src.cli.runner prepare-text-normalize --prep_name=$prep_name --text_name=$text_name
python -m src.cli.runner prepare-text-to-ipa --prep_name=$prep_name --text_name=$text_name

## Chinese
export text="examples/chn/thchs.txt"
export text_name="chn-thchs"
python -m src.cli.runner prepare-text-add --filepath=$text --prep_name=$prep_name --text_name=$text_name --lang=CHN
python -m src.cli.runner prepare-text-normalize --prep_name=$prep_name --text_name=$text_name
python -m src.cli.runner prepare-text-to-ipa --prep_name=$prep_name --text_name=$text_name

export text="examples/chn/north.txt"
export text_name="chn-north"
python -m src.cli.runner prepare-text-add --filepath=$text --prep_name=$prep_name --text_name=$text_name --lang=CHN
python -m src.cli.runner prepare-text-normalize --prep_name=$prep_name --text_name=$text_name
python -m src.cli.runner prepare-text-to-ipa --prep_name=$prep_name --text_name=$text_name

## German
export text="examples/ger/nord.txt"
export text_name="ger-nord"
python -m src.cli.runner prepare-text-add --filepath=$text --prep_name=$prep_name --text_name=$text_name --lang=GER
python -m src.cli.runner prepare-text-normalize --prep_name=$prep_name --text_name=$text_name
python -m src.cli.runner prepare-text-to-ipa --prep_name=$prep_name --text_name=$text_name

## English
export text="examples/en/coma.txt"
export text_name="eng-coma"
python -m src.cli.runner prepare-text-add --filepath=$text --prep_name=$prep_name --text_name=$text_name --lang=ENG
python -m src.cli.runner prepare-text-normalize --prep_name=$prep_name --text_name=$text_name
python -m src.cli.runner prepare-text-to-ipa --prep_name=$prep_name --text_name=$text_name

export text="examples/en/rainbow.txt"
export text_name="eng-rainbow"
python -m src.cli.runner prepare-text-add --filepath=$text --prep_name=$prep_name --text_name=$text_name --lang=ENG
python -m src.cli.runner prepare-text-normalize --prep_name=$prep_name --text_name=$text_name
python -m src.cli.runner prepare-text-to-ipa --prep_name=$prep_name --text_name=$text_name

export text="examples/en/stella.txt"
export text_name="eng-stella"
python -m src.cli.runner prepare-text-add --filepath=$text --prep_name=$prep_name --text_name=$text_name --lang=ENG
python -m src.cli.runner prepare-text-normalize --prep_name=$prep_name --text_name=$text_name
python -m src.cli.runner prepare-text-to-ipa --prep_name=$prep_name --text_name=$text_name

export text="examples/en/democritus_v2.txt"
export text_name="eng-democritus_v2"
python -m src.cli.runner prepare-text-add --filepath=$text --prep_name=$prep_name --text_name=$text_name --lang=ENG
python -m src.cli.runner prepare-text-normalize --prep_name=$prep_name --text_name=$text_name
python -m src.cli.runner prepare-text-to-ipa --prep_name=$prep_name --text_name=$text_name

export text="examples/en/north.txt"
export text_name="eng-north"
python -m src.cli.runner prepare-text-add --filepath=$text --prep_name=$prep_name --text_name=$text_name --lang=ENG
python -m src.cli.runner prepare-text-normalize --prep_name=$prep_name --text_name=$text_name
python -m src.cli.runner prepare-text-to-ipa --prep_name=$prep_name --text_name=$text_name
