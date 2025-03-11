import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt

def draw_bounding_boxes_from_csv(image_dir_path, csv_path):
    output_folder = "output_images/"
    os.makedirs(output_folder, exist_ok=True)

    image_name = None
    
    df = pd.read_csv(csv_path)
    grouped = df.groupby('image_name')
    
    for image_name, group in grouped:
         
        image_name += ".JPG"
        image_path = os.path.join(image_dir_path, image_name)
        image = cv2.imread(image_path)
    
        for _, row in df.iterrows():
            xmin, ymin, xmax, ymax = map(int, [row['xmin'], row['ymin'], row['xmax'], row['ymax']])
            confidence = row.get('iou', -1)
            label = row.get('scroll_number', "scroll")

            # Draw rectangle
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            # Draw label and confidence
            text = f"{label}: {confidence:.2f}"
            cv2.putText(image, text, (xmin, max(ymin - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 
                        2, (0, 0, 255), 3)  # Bigger font
        
        output_path = os.path.join(output_folder, image_name)    
        cv2.imwrite(output_path, image)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    image_dir_path = "saraay@post.jce.ac.il"
    csv_path = "yolov8n_results.csv"
    
    draw_bounding_boxes_from_csv(image_dir_path, csv_path)
    