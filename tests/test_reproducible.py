from loguru import logger

from app.utils.reproducible import Reproducible


def test():
    rm = Reproducible(seed=369)
    logger.debug("rm: {}", rm)
    res = rm.model_dump(mode='json')
    v1 = rm.rng.random()
    logger.debug("v1: {}", v1)
    rm2 = Reproducible(**res)
    v2 = rm2.rng.random()
    logger.debug("v2: {}", v2)

    assert v1 == v2
