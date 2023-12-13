from yacs.config import CfgNode as CN

_C = CN()
_C.NAME = ""
_C.SEED = 1
_C.SC_SEND = False

# -----------------------------------------------------------------------------
# PATH
# -----------------------------------------------------------------------------

_C.PATH = CN()
_C.PATH.DATASETS = "../datasets"
_C.PATH.OUTPUTS = "../output"
_C.PATH.PRE_TRAINED = "../pretrained/clip"


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# CUHK-PEDES, ICFG-PEDES
_C.DATASETS.NAME = ""
# val, test
_C.DATASETS.VAL = 'val'


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.IMS_PER_ID = 4
# random, identity
_C.DATALOADER.SAMPLER = 'random'
_C.DATALOADER.TEXT_REMOVE_COLOR = False
_C.DATALOADER.TEXT_REMOVE_COLOR_PRO = (0.5, 0.2, 0.1)
_C.DATALOADER.REMOVE_COLOR_ON_SENTENCE = False
_C.DATALOADER.SEN_DROP_PROB = 0.5
_C.DATALOADER.PER_SEN_MAX = 2
_C.DATALOADER.RANDOM_REMOVE_SET = (1., 2)
_C.DATALOADER.MOD_PERCENT_PER_SEN = 0.15
_C.DATALOADER.BERT_DROP_PROB = 0.8
_C.DATALOADER.BERT_CHANGE_PROB = 0.1
_C.DATALOADER.BERT_REMOVE = False
_C.DATALOADER.MASK_COLOR_TYPE = 'random_mask_BERT_color'
_C.DATALOADER.MASK_COLOR = False
_C.DATALOADER.MASK_PROB_PER_SEN = 0.15
_C.DATALOADER.MASK_PROB = 0.8
_C.DATALOADER.TEXT_ENCODE_TYPE = 'default'
_C.DATALOADER.TEXT_ENCODE_TYPE_2 = 'default'
_C.DATALOADER.DROP_PROB_PER_SEN = 0.2
_C.DATALOADER.CHANGE_SEN_PROB = 0.5
_C.DATALOADER.MAIN_MOD_PERCENT_PER_SEN = 0.5
_C.DATALOADER.MAIN_MASK_PROB_PER_SEN = 0.4
_C.DATALOADER.MAIN_DROP_PROB_PER_SEN = 0.4
_C.DATALOADER.MAIN_TEXT_ENCODE_TYPE = 'default'

# simple, link, one_per_sen
_C.DATALOADER.REMOVE_COLOR_TYPE = 'link'


# -----------------------------------------------------------------------------
# Input
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.SIZE = (384, 128)
_C.INPUT.USE_AUG = True


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.VISUAL_MODEL = "ViT-B/16"
_C.MODEL.TEXTUAL_MODEL = "ViT-B/16"
_C.MODEL.USE_FP16 = True
_C.MODEL.TEXTUAL_MODEL_FREEZE_LAYER = 0
_C.MODEL.FREEZE_CONV1 = False
_C.MODEL.TEXT_DROPOUT = 0.


# -----------------------------------------------------------------------------
# Embedding
# -----------------------------------------------------------------------------
_C.MODEL.EMBEDDING = CN()
_C.MODEL.EMBEDDING.FEATURE_SIZE = 512
_C.MODEL.EMBEDDING.EMBED_HEAD = 'default'
_C.MODEL.EMBEDDING.EPSILON = 0.1
_C.MODEL.EMBEDDING.LAYERS = 1
_C.MODEL.EMBEDDING.CROSS_SELF_ATTENTION_MASK = True


# -----------------------------------------------------------------------------
# LOSS
# -----------------------------------------------------------------------------

_C.LOSS = CN()
_C.LOSS.BCL_ALPHA = 0.6
_C.LOSS.BCL_SCALE_ALPHA = 10
_C.LOSS.BCL_BETA = 0.4
_C.LOSS.BCL_SCALE_BETA = 40
_C.LOSS.BCL_GAMA = 0.45
_C.LOSS.BCL_SCALE_GAMA = 4
_C.LOSS.MARGIN_LOSS_MARGIN = 0.2
_C.LOSS.MARGIN_LOSS_POS_MARGIN = 0.2
_C.LOSS.MARGIN_LOSS_SM_MARGIN = 0.05
_C.LOSS.MARGIN_LOSS_SCALE = 0.5
_C.LOSS.TAU = 0.02
_C.LOSS.LOSSES = ('BCL', 'id_loss', )
_C.LOSS.MARGIN_LOSS_SM_SCALE = 1.0


# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.IMS_PER_BATCH = 16
_C.SOLVER.NUM_EPOCHS = 100
_C.SOLVER.CHECKPOINT_EPOCH = (50, 60, )
_C.SOLVER.EVALUATE_PERIOD = 1

_C.SOLVER.OPTIMIZER = "Adam"
_C.SOLVER.BASE_LR = 1e-5
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.INIT_FACTOR = 5.0

_C.SOLVER.WEIGHT_DECAY = 5e-4
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0

_C.SOLVER.ADAM_ALPHA = 0.9
_C.SOLVER.ADAM_BETA = 0.999
_C.SOLVER.SGD_MOMENTUM = 0.9

_C.SOLVER.LRSCHEDULER = "cosine"

_C.SOLVER.WARMUP_FACTOR = 0.1
_C.SOLVER.WARMUP_EPOCHS = 5
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (500,)

_C.SOLVER.POWER = 0.9
_C.SOLVER.TARGET_LR = 0.

_C.SOLVER.LOG_PERIOD = 100


_C.SOLVER.CROSS_FACTOR = 5.0
_C.SOLVER.START_LR = 1.0e-6


# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 16