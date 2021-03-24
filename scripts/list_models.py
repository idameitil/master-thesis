import os
import pickle

def save_obj(obj, name):
    with open('/home/ida/master-thesis/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open('/home/ida/master-thesis/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

succeded_models = load_obj("succeded_models")
print(succeded_models)
print(len(succeded_models["1"]["positives"])+len(succeded_models["2"]["positives"])+len(succeded_models["3"]["positives"])+len(succeded_models["4"]["positives"])+len(succeded_models["5"]["positives"]))
print(len(succeded_models["1"]["negatives"])+len(succeded_models["2"]["negatives"])+len(succeded_models["3"]["negatives"])+len(succeded_models["4"]["negatives"])+len(succeded_models["5"]["negatives"]))
# for partition in range(1,6):
#     partition = str(partition)
#     for binder in ("positives", "negatives"):
#         succeded_models[partition][binder] = \
#             [el.replace('_model', '') for el in succeded_models[partition][binder]]
#
# save_obj(succeded_models, "succeded_models")