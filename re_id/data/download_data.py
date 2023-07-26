import os
import requests 
import shutil

from dataclasses import dataclass
from zipfile import ZipFile

@dataclass
class Configuration:
    link: str = "https://github.com/DeepSportRadar/player-reidentification-challenge/archive/refs/heads/master.zip"
    md5: str = '05715857791e2e88b2f11e4037fbec7d'
    path: str = "../data"
    
def download_zip(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

#----------------------------------------------------------------------------------------------------------------------#  
# Config                                                                                                               #
#----------------------------------------------------------------------------------------------------------------------# 
config = Configuration()

#----------------------------------------------------------------------------------------------------------------------#  
# Custom Dataset Initialization                                                                                                               #
#----------------------------------------------------------------------------------------------------------------------# 
if not os.path.exists(config.path):
    print("Custom dataset path is not found. Initializing...")
    os.makedirs(config.path)
    os.mkdir(os.path.join(config.path, "custom_dataset"))
    os.mkdir(os.path.join(config.path, "custom_dataset", "gallery"))
    os.mkdir(os.path.join(config.path, "custom_dataset", "query"))
    os.mkdir(os.path.join(config.path, "custom_dataset", "training"))
else:
    print("Custom dataset path is already established.")

#----------------------------------------------------------------------------------------------------------------------#  
# Download                                                                                                             #
#----------------------------------------------------------------------------------------------------------------------#
if not os.path.isfile("{}/synergyreid_data.zip".format(config.path)):
    print("Downloading demo dataset...")

    download_zip(url=config.link,
                 save_path="{}/reid_challenge.zip".format(config.path),
                 chunk_size=128)

    path_in_zip = "player-reidentification-challenge-master/baseline/data/synergyreid/raw/synergyreid_data.zip"
         
    with ZipFile("{}/reid_challenge.zip".format(config.path)) as z:

        z.extract(path_in_zip,
                  config.path)
        
    shutil.move("{}/{}".format(config.path, path_in_zip), "{}/synergyreid_data.zip".format(config.path))
    
    shutil.rmtree("{}/player-reidentification-challenge-master".format(config.path)) 
    os.remove("{}/reid_challenge.zip".format(config.path))
else:
    print("Demo dataset is already downloaded. Moving to next step...")

#----------------------------------------------------------------------------------------------------------------------#  
# Unzip & Clean Up                                                                                                             #
#----------------------------------------------------------------------------------------------------------------------#  
zip_file = "{}/synergyreid_data.zip".format(config.path)
    
print("Extracting zip file...")
with ZipFile(zip_file) as z:
    z.extractall(path=config.path)

shutil.rmtree(f"{config.path}/__MACOSX")

print("Extraction Completed!\n")