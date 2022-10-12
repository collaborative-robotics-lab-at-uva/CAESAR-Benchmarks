# Base VL model key list
# base_model_vl_dual_encoder
# base_model_vl_clip
# base_model_vl_lxmert
# base_model_vl_visual_bert


# egoview_image_tag = 'egoview_image'
# exoview_image_tag = 'exoview_image'
# topview_image_tag = 'topview_image'
egoview_image_tag = 'ego_view_image'
exoview_image_tag = 'exo_view_image'
topview_image_tag = 'top_view_image'
instruction_valid_task_tag = 'is_contrastive'
ambiguity_recognition_task_tag = 'is_instruction_ambiguous'

task_name_to_id = {f'{egoview_image_tag}': 1,
                f'{exoview_image_tag}': 2,
                f'{topview_image_tag}': 3,
                f'{instruction_valid_task_tag}': 4,
                f'{ambiguity_recognition_task_tag}': 5}

# image scale 0.35
# image_width_normalization = 252.0
# image_height_normalization = 168.0

# image scale 1.0
image_width_normalization = 720.0
image_height_normalization = 480.0

# image scale 0.5
image_width_normalization = 360.0
image_height_normalization = 240.0

# image_width_normalization = 1.0
# image_height_normalization = 1.0

bbox_format_xywh = 'xywh'

dataset_split_tag = 'split'
train_dataset_tag = 'train'
test_dataset_tag = 'test'

image_width = 224
image_height = 224
image_channels = 3

attention_type_sum = 'sum'

# Model tag
mm_attn_encoder = 'mm_attn_encoder'
hamlet_encoder = 'hamlet_encoder'
keyless_encoder = 'keyless_encoder'


#Logging config
tbw_train_loss = 'Loss/Train'
tbw_valid_loss = 'Loss/Valid'
tbw_test_loss = 'Loss/Test'

tbw_train_acc = 'Accuracy/Train'
tbw_valid_acc = 'Accuracy/Valid'
tbw_test_acc = 'Accuracy/Test'

tbw_train_f1 = 'F1/Train'
tbw_valid_f1 = 'F1/Valid'
tbw_test_f1 = 'F1/Test'

tbw_train_precision = 'Precision/Train'
tbw_valid_precision = 'Precision/Valid'
tbw_test_precision = 'Precision/Test'

tbw_train_recall = 'Recall/Train'
tbw_valid_recall = 'Recall/Valid'
tbw_test_recall = 'Recall/Test'

