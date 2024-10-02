from segment_anything import SamPredictor, sam_model_registry
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from transformers import AutoProcessor, AutoModel
import numpy as np

class SAMIoU:
    '''
    Example:
    def evaluate_one_image(image_np, bboxes):
        sam_iou.set_image(image_np)
        ious = []
        object_size = []
        for input_box in bboxes:
            input_box = input_box.astype(np.int32)
            object_size.append((input_box[2] - input_box[0]) * (input_box[3] - input_box[1]))
            sam_mask1, score1 = sam_iou.get_mask_from_bbox(input_box)
            sam_box1 = sam_iou.get_bbox_from_mask(sam_mask1)
            iou1 = sam_iou.compute_iou(input_box, sam_box1)
            ious.append(iou1)
        return np.mean(ious), ious, object_size
    '''
    def __init__(self):
        sam = sam_model_registry["default"](checkpoint="tmp/sam_vit_h_4b8939.pth").cuda()
        self.predictor = SamPredictor(sam)

    def set_image(self, image):
        self.predictor.set_image(image)

    def get_mask_from_bbox(self, bbox):
        input_box = np.array(bbox)
        mask, score, logit = self.predictor.predict(
            box=input_box, 
            multimask_output=False,
        )
        return mask[0], score[0]

    def get_bbox_from_mask(self, mask):
        # mask: np.array binary mask of shape (H, W), get a out bounded box
        rows = np.any(mask, axis=1)
        first_row, last_row = np.where(rows)[0][[0, -1]]
        cols = np.any(mask, axis=0)
        first_col, last_col = np.where(cols)[0][[0, -1]]
        box = [first_col, first_row, last_col, last_row]
        return box

    def get_mask_from_points(self, points):
        input_points = np.array(points)
        input_labels = np.ones(input_points.shape[0], dtype=np.int32)
        mask, score, logit = self.predictor.predict(
            point_coords=input_points, 
            point_labels=input_labels,
            multimask_output=False,
        )
        return mask[0], score[0]

    def get_points_from_mask(self, mask, seed=None, num_points=3):
        np.random.seed(seed)
        ys, xs = np.where(mask > 0)
        selected_indices = np.random.choice(len(xs), num_points, replace=False)
        return list(zip(xs[selected_indices], ys[selected_indices]))

    def compute_iou(self, bbox1, bbox2):
        x1, y1, x2, y2 = bbox1
        x1_, y1_, x2_, y2_ = bbox2

        x_left = max(x1, x1_)
        y_top = max(y1, y1_)
        x_right = min(x2, x2_)
        y_bottom = min(y2, y2_)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = (x2 - x1) * (y2 - y1)
        bbox2_area = (x2_ - x1_) * (y2_ - y1_)

        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        return iou
    

class CropCLIPScore:
    '''
    Example:
    def evaluate_one_image(image_np, bboxes, labels):
        all_simi = []
        object_size = []
        for input_box, label in zip(bboxes, labels):
            input_box = input_box.astype(np.int32)
            simi = clip_scorer.compute_score_wbbox(image_np, label, input_box)
            if simi is not None:
                all_simi.append(simi)
                object_size.append((input_box[2] - input_box[0]) * (input_box[3] - input_box[1]))
        if len(all_simi) > 0:
            return np.mean(all_simi), all_simi, object_size
        return None, None, None
    '''
    def __init__(self, model_name = "openai/clip-vit-base-patch32"):
        self.model = CLIPModel.from_pretrained(model_name).cuda()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def compute_score_wbbox(self, image_np, region_label, bbox):
        image = Image.fromarray(image_np)
        # Crop the image
        image_cropped = image.crop(bbox)

        # Preprocess the image and text
        # inputs = self.processor(text=[region_label], images=image_cropped, return_tensors="pt", truncation=True)
        encoding = self.processor.tokenizer([region_label], return_tensors="pt", padding=True, truncation=True)
        try:
            image_features = self.processor.image_processor(images=image_cropped, return_tensors="pt")
        except:
            return None

        encoding['pixel_values'] = image_features.pixel_values
        inputs = {
            k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in encoding.items()
        }

        # Compute the similarity score
        with torch.no_grad():
            outputs = self.model(**inputs)
            similarity = outputs.logits_per_image[0][0].item()

        return similarity

class CropPickScore:
    '''
    Example:
    def evaluate_one_image(image_np, bboxes, labels):
        all_simi = []
        object_size = []
        for input_box, label in zip(bboxes, labels):
            input_box = input_box.astype(np.int32)
            simi = clip_scorer.compute_score_wbbox(image_np, label, input_box)
            if simi is not None:
                all_simi.append(simi)
                object_size.append((input_box[2] - input_box[0]) * (input_box[3] - input_box[1]))
        if len(all_simi) > 0:
            return np.mean(all_simi), all_simi, object_size
        return None, None, None
    '''
    def __init__(self, model_name = "yuvalkirstain/PickScore_v1", processor_name="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"):
        self.model = AutoModel.from_pretrained(model_name).eval().cuda()
        self.processor = AutoProcessor.from_pretrained(processor_name)

    @torch.no_grad()
    def compute_score_wbbox(self, image_np, region_label, bbox):
        image = Image.fromarray(image_np)
        # Crop the image
        image_cropped = image.crop(bbox)

        # preprocess
        try:
            image_inputs = self.processor(
                images=[image_cropped],
                return_tensors="pt",
            ).to('cuda')
        except:
            return None
        
        text_inputs = self.processor(
            text=region_label,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to('cuda')

        image_embs = self.model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

        return scores.item()