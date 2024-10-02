import os
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import imageio
from PIL import Image, ImageEnhance
import textwrap
from torchvision.io import read_video

def find_nearest_Nx(size, N=32):
    return int(np.ceil(size / N) * N)

def load_image_as_tensor(image_path, image_size):
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    image = cv2.imread(image_path)[..., ::-1]
    try:
        image = cv2.resize(image, image_size)
    except Exception as e:
        print(e)
        print(image_path)

    image = torch.from_numpy(np.array(image).transpose(2, 0, 1)) / 255.
    return image

def show_image(image):
    if len(image.shape) == 4:
        image = image[0]
    plt.imshow(image.permute(1, 2, 0).detach().cpu().numpy())
    plt.show()

def extract_video(video_path, save_dir, sampling_fps, skip_frames=0):
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_skip = int(cap.get(cv2.CAP_PROP_FPS) / sampling_fps)
    frame_count = 0
    save_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count < skip_frames:  # skip the first N frames
            frame_count += 1
            continue
        if (frame_count - skip_frames) % frame_skip == 0:
            # Save the frame as an image file if it doesn't already exist
            save_path = os.path.join(save_dir, f"frame{save_count:04d}.jpg")
            save_count += 1
            if not os.path.exists(save_path):
                cv2.imwrite(save_path, frame)
        frame_count += 1
    cap.release()
    cv2.destroyAllWindows()

def concatenate_frames_to_video(frame_dir, video_path, fps):
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    # Get the list of frame file names in the directory
    frame_files = [f for f in os.listdir(frame_dir) if f.startswith("frame")]
    # Sort the frame file names in ascending order
    frame_files.sort()
    # Load the first frame to get the frame size
    frame = cv2.imread(os.path.join(frame_dir, frame_files[0]))
    height, width, _ = frame.shape
    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    # Loop through the frame files and add them to the video
    for frame_file in frame_files:
        frame_path = os.path.join(frame_dir, frame_file)
        frame = cv2.imread(frame_path)
        out.write(frame)
    # Release the video writer
    out.release()


# def images_to_gif(images, filename, fps):
#     os.makedirs(os.path.dirname(filename), exist_ok=True)
#     # Normalize to 0-255 and convert to uint8
#     images = [(img * 255).astype(np.uint8) if img.dtype == np.float32 else img for img in images]
#     images = [Image.fromarray(img) for img in images]
#     imageio.mimsave(filename, images, duration=1 / fps, loop=0)
def make_gif(frames, filename, fps=8, rescale=0.5):
    if os.path.dirname(filename) != '':
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    # resize frames
    if rescale is not None:
        frames = [Image.fromarray(frame) for frame in frames]
        frames = [frame.resize((int(frame.width * rescale), int(frame.height * rescale))) for frame in frames]
        frames = [np.array(frame) for frame in frames]
    imageio.mimsave(filename, frames, duration=1000 / fps, loop=0)

def load_gif(image_path):
    import imageio
    gif = imageio.get_reader(image_path)
    np_images = np.array([frame[..., :3] for frame in gif])
    return np_images

def add_text_to_frame(frame, text, font_scale=1, thickness=2, color=(0, 0, 0), bg_color=(255, 255, 255), max_width=30):
    """
    Add text to a frame.
    """
    # Make a copy of the frame
    frame_with_text = np.copy(frame)
    # Choose font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Split text into lines if it's too long
    lines = textwrap.wrap(text, width=max_width)
    # Get total text height
    total_text_height = len(lines) * (thickness * font_scale + 10) + 60 * font_scale
    # Create an image filled with the background color, having enough space for the text
    text_bg_img = np.full((int(total_text_height), frame.shape[1], 3), bg_color, dtype=np.uint8)
    # Put each line on the text background image
    y = 0
    for line in lines:
        text_size, _ = cv2.getTextSize(line, font, font_scale, thickness)
        text_x = (text_bg_img.shape[1] - text_size[0]) // 2
        y += text_size[1] + 10
        cv2.putText(text_bg_img, line, (text_x, y), font, font_scale, color, thickness)
    # Append the text background image to the frame
    frame_with_text = np.vstack((frame_with_text, text_bg_img))
    
    return frame_with_text

def add_text_to_gif(numpy_images, text, **kwargs):
    """
    Add text to each frame of a gif.
    """
    # Iterate over frames and add text to each frame
    frames_with_text = []
    for frame in numpy_images:
        frame_with_text = add_text_to_frame(frame, text, **kwargs)
        frames_with_text.append(frame_with_text)

    # Convert the list of frames to a numpy array
    numpy_images_with_text = np.array(frames_with_text)
    
    return numpy_images_with_text

def pad_images_to_same_height(images):
    """
    Pad images to the same height.
    """
    # Find the maximum height
    max_height = max(img.shape[0] for img in images)
    
    # Pad each image to the maximum height
    padded_images = []
    for img in images:
        pad_height = max_height - img.shape[0]
        padded_img = cv2.copyMakeBorder(img, 0, pad_height, 0, 0, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        padded_images.append(padded_img)
    
    return padded_images

def concatenate_gifs(gifs):
    """
    Concatenate gifs.
    """
    # Ensure that all gifs have the same number of frames
    min_num_frames = min(gif.shape[0] for gif in gifs)
    gifs = [gif[:min_num_frames] for gif in gifs]
    
    # Concatenate each frame
    concatenated_gifs = []
    for i in range(min_num_frames):
        # Get the i-th frame from each gif
        frames = [gif[i] for gif in gifs]
        
        # Pad the frames to the same height
        padded_frames = pad_images_to_same_height(frames)
        
        # Concatenate the padded frames
        concatenated_frame = np.concatenate(padded_frames, axis=1)
        
        concatenated_gifs.append(concatenated_frame)

    return np.array(concatenated_gifs)

def stack_gifs(gifs):
    '''vertically stack gifs'''
    min_num_frames = min(gif.shape[0] for gif in gifs)
    stacked_gifs = []

    for i in range(min_num_frames):
        frames = [gif[i] for gif in gifs]
        stacked_frame = np.concatenate(frames, axis=0)
        stacked_gifs.append(stacked_frame)

    return np.array(stacked_gifs)


def brightness_and_saturation_shift(image, brightness_factor=(1.0, 1.0, 1.0), saturation_factor=1.0):
    # Check the type of the input and convert to a PIL Image if necessary
    if isinstance(image, str):
        # Load the image from the file path
        img = Image.open(image)
    elif isinstance(image, np.ndarray):
        # Convert the NumPy array to a PIL Image
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Assume the image is in 0-1 range
            image = (image * 255).astype(np.uint8)
        img = Image.fromarray(image)
    elif isinstance(image, Image.Image):
        img = image
    else:
        raise ValueError("Unsupported image format")
    
    # Apply brightness shift to each channel
    channels = img.split()
    enhanced_channels = []
    for i, channel in enumerate(channels):
        enhancer = ImageEnhance.Brightness(channel)
        enhanced_channel = enhancer.enhance(brightness_factor[i])
        enhanced_channels.append(enhanced_channel)
    
    # Merge the channels back together
    brightened_img = Image.merge(img.mode, enhanced_channels)
    
    # Apply saturation shift to each channel
    saturated_img = ImageEnhance.Color(brightened_img).enhance(saturation_factor)
    
    # Convert the PIL Image to a NumPy array and normalize to 0-1 range
    final_img_array = np.asarray(saturated_img).astype(np.float32) / 255.0
    
    return final_img_array


'''
Usage:
video, fps = load_video_and_fps('data/car-turn.mp4', 0)
cropped_video = crop_and_resize_video(video, HEIGHT, WIDTH)
'''
def load_video_and_fps(video_path, start_sec, end_sec=None):
    video, _, fps = read_video(video_path, pts_unit='sec', start_pts=start_sec, end_pts=end_sec, output_format='TCHW')
    return video, fps['video_fps']

def crop_and_resize_video(video, height, width):
    ratio = height / width
    current_ratio = video.shape[2] / video.shape[3]
    if ratio > current_ratio:
        # crop width
        crop_height = video.shape[2]
        crop_width = video.shape[2] / ratio
        crop_width = int(crop_width - crop_width % 2)
    else:
        # crop height
        crop_height = video.shape[3] * ratio
        crop_height = int(crop_height - crop_height % 2)
        crop_width = video.shape[3]

    crop_height = min(crop_height, video.shape[2])
    crop_width = min(crop_width, video.shape[3])

    center_height, center_width = video.shape[2] // 2, video.shape[3] // 2

    video = video[:, :, center_height - crop_height // 2:center_height + crop_height // 2,
            center_width - crop_width // 2:center_width + crop_width // 2]
    
    video = torch.nn.functional.interpolate(video, size=(height, width), mode='nearest') # bilinear has problem when HEIGHT > WIDTH
    return video

