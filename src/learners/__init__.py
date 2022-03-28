from .q_learner import QLearner
from .qr_learner import QRLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .qatten_learner import QattenLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .max_q_learner import MAXQLearner


REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["qr_learner"] = QRLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["qatten_learner"] = QattenLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner
REGISTRY["max_q_learner"] = MAXQLearner

