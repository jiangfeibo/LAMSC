import os

import SKB
import ASI
import SC_with_ASC
if __name__ == '__main__':
    dataset = "example"
    # image segment based on SKB
    raw_images_path = f"data/raw_images/{dataset}" # path of raw images
    for img in os.listdir(raw_images_path):
        img_path = os.path.join(raw_images_path,img)
        SKB.SKB_with_auto(img_path,device="cpu",dataset=dataset)
    # semantic-aware image generation based on ASI
    segment_images_path = f"data/segments/{dataset}" # path of segment images
    ASI.semantic_aware_images_generation(segment_images_path,dataset=dataset)

    # image SC based on SC with ASC
    sem_aware_images_path = "data/semantic-aware_images/"
    SC_with_ASC.data_transmission(sem_aware_images_path,dataset=dataset)


