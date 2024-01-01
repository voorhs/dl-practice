python3 parse_persona_chat.py --task $1
python3 sentence_encoding.py --task $1
python3 dgac_clustering.py --task $1
python3 linking_prediction.py --task $1
python3 response_selection.py --task $1
python3 acc_of_response_selector.py --task $1