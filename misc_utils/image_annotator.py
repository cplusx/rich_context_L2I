from PIL import Image, ImageDraw, ImageFont
import numpy as np
import textwrap

'''
To find install fonts on Linux, use
find /usr/share/fonts/ -name "*.ttf"

Usage:

# image should be np.array of shape (H, W, 3)
image_with_bboxes = annotator.add_bboxes(image, bboxes, labels)
image_with_masks = annotator.add_masks(image, masks, labels)
'''

class ImageAnnotator:
    def __init__(self):
        # Default font, will be dynamically adjusted
        self.font = ImageFont.load_default()
        self.text_size=30
        self._line_spacing = 4

    def add_bboxes(self, image, bboxes, labels, color=None, linewidth=None):
        self._check_input_type(image)
        image = self._convert_image(image)
        draw = ImageDraw.Draw(image)

        for box_idx, (bbox, label) in enumerate(zip(bboxes, labels)):
            if label == '':
                continue
            if isinstance(color, list):
                color_ = color[box_idx]
            else:
                color_ = self._get_random_color() if color is None else color
            self._adjust_font(image)
            draw.rectangle([(bbox[0], bbox[1]), (bbox[2], bbox[3])], outline=color_, width=linewidth or self._line_width(image))

            if label != ' ':
                max_width = self._max_text_width(draw=draw, label=label)
                wrapped_text = textwrap.fill(label, width=max_width)
                text_size = self._multiline_text_size(draw, wrapped_text)
                text_position = (bbox[0], max(bbox[1] - text_size[1], 0))
                draw.multiline_text(text_position, wrapped_text, fill=color_, font=self.font)

        return self._convert_back(image)

    def _multiline_text_size(self, draw, text):
        lines = text.split('\n')
        width = max(draw.textlength(line, font=self.font) for line in lines)
        height = self.text_size * len(lines) + (len(lines) - 1) * self._line_spacing
        return width, height

    def _max_text_width(self, draw, label):
        # This function now calculates the maximum text width based on the label's length
        # and the current font size. It returns the maximum number of characters per line.
        max_line_length = 0
        for word in label.split():
            word_length = draw.textlength(word, font=self.font)
            if word_length > max_line_length:
                max_line_length = word_length
        return max_line_length

    def add_masks(self, image, masks, labels):
        self._check_input_type(image)
        original_image = self._convert_image(image).convert('RGBA')
        draw = ImageDraw.Draw(original_image)

        for mask, label in zip(masks, labels):
            color = self._get_random_color(True)
            rgba_mask = self._create_rgba_mask(mask, color)
            mask_image = Image.fromarray(rgba_mask)

            # Ensure mask_image is the same size as original_image
            mask_image = mask_image.resize(original_image.size)

            # Blend the images
            blended_image = Image.blend(original_image, mask_image, 0.5)
            original_image.paste(blended_image, (0, 0), mask_image)

            # Calculate the position for the label
            x, y = np.where(mask != 0)
            if len(x) == 0 or len(y) == 0: 
                continue
            text_position = (np.mean(y), np.mean(x))
            text_color = 'white' if np.mean(color[:3]) < 128 else 'black'
            draw.text(text_position, label, fill=text_color, font=self.font)

        return self._convert_back(original_image)

    def _create_rgba_mask(self, mask, color):
        # Convert a 2D mask to an RGBA mask compatible with the provided color
        rgba_mask = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        for i in range(3):
            rgba_mask[:, :, i] = mask * color[i]
        rgba_mask[:, :, 3] = mask * color[3]  # Alpha channel
        return rgba_mask

    def _convert_image(self, image):
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        return Image.fromarray(image)

    def _convert_back(self, image):
        if self.original_dtype == np.float32 or self.original_dtype == np.float64:
            return np.array(image) / 255.0
        return np.array(image)

    def _get_random_color(self, for_mask=False):
        color = tuple(np.random.choice(range(256), size=3))
        return color if not for_mask else (color[0], color[1], color[2], 128)  # Add alpha for transparency

    def _check_input_type(self, image):
        self.original_dtype = image.dtype

    def _adjust_font(self, image):
        # Adjust font size based on image size
        font_size = max(min(image.size) // 40, self.text_size)  # example scaling, can be adjusted
        self.font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=font_size)

    def _line_width(self, image):
        # Adjust line width based on image size
        return max(min(image.size) // 200, 1)

# Example usage:
# annotator = ImageAnnotator()
# annotated_image = annotator.add_bboxes(image, bboxes, labels)
# annotated_image_with_masks = annotator.add_masks(image, masks, labels)
