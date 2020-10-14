# Inference

## Capslock
source /datasets/code/tacotron2/configs/envs/caps.sh

## Phil
source /home/stefan/tacotron2/configs/envs/phil.sh

export prep_name="ljs_ipa"
export prep_name="thchs_ipa"
export prep_name="thchs_ljs_ipa"
export prep_name="arctic_ipa_22050"
export prep_name="libritts_ipa_22050"

## Quick-Test
export text="examples/quick-test.txt"
export text_name="quick-test"
python -m src.cli.runner prepare-text-add --filepath=$text --prep_name=$prep_name --text_name=$text_name --lang=ENG
python -m src.cli.runner prepare-text-normalize --prep_name=$prep_name --text_name=$text_name
python -m src.cli.runner prepare-text-to-ipa --prep_name=$prep_name --text_name=$text_name



## IPA
export text="examples/ipa/north_sven.txt"
export text_name="ipa-north_sven"
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

export text="examples/chn/north_orig.txt"
export text_name="chn-north-orig"
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

export text="examples/ger/example.txt"
export text_name="ger-example"
python -m src.cli.runner prepare-text-add --filepath=$text --prep_name=$prep_name --text_name=$text_name --lang=GER
python -m src.cli.runner prepare-text-normalize --prep_name=$prep_name --text_name=$text_name
python -m src.cli.runner prepare-text-to-ipa --prep_name=$prep_name --text_name=$text_name

## English
export text="examples/en/wolf.txt"
export text_name="eng-wolf"
python -m src.cli.runner prepare-text-add --filepath=$text --prep_name=$prep_name --text_name=$text_name --lang=ENG
python -m src.cli.runner prepare-text-normalize --prep_name=$prep_name --text_name=$text_name
python -m src.cli.runner prepare-text-to-ipa --prep_name=$prep_name --text_name=$text_name

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

export text="examples/en/democritus.txt"
export text_name="eng-democritus"
python -m src.cli.runner prepare-text-add --filepath=$text --prep_name=$prep_name --text_name=$text_name --lang=ENG
python -m src.cli.runner prepare-text-normalize --prep_name=$prep_name --text_name=$text_name
python -m src.cli.runner prepare-text-to-ipa --prep_name=$prep_name --text_name=$text_name

export text="examples/en/north.txt"
export text_name="eng-north"
python -m src.cli.runner prepare-text-add --filepath=$text --prep_name=$prep_name --text_name=$text_name --lang=ENG
python -m src.cli.runner prepare-text-normalize --prep_name=$prep_name --text_name=$text_name
python -m src.cli.runner prepare-text-to-ipa --prep_name=$prep_name --text_name=$text_name



# Update Inference Map

python -m src.cli.runner prepare-inference-map \
  --prep_name=$prep_name \
  --template_map="maps/inference/eng_ipa.json"

python -m src.cli.runner prepare-inference-map \
  --prep_name=$prep_name \
  --template_map="maps/inference/chn_ipa.json"

python -m src.cli.runner prepare-text-automap \
  --prep_name=$prep_name \
  --text_name=$text_name
