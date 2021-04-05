import logging


def set_logger(LOGGING_FOLDER, format_time, computer_name, CMD_LOG_LEVEL):
    # 设置Logging
    FILE_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
    FILE_DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logger = logging.getLogger()
    logger.setLevel(logging.NOTSET)

    # 使用FileHandler输出到文件
    file_formatter = logging.Formatter(
        FILE_LOG_FORMAT, datefmt=FILE_DATE_FORMAT)
    fh = logging.FileHandler(
        LOGGING_FOLDER+r'/'+format_time+'_'+computer_name + '.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(file_formatter)

    # 修改自带的输出到控制台的handler
    cmd_formatter = logging.Formatter("%(message)s")
    ch = logger.handlers[0]
    ch.setLevel(CMD_LOG_LEVEL)
    ch.setFormatter(cmd_formatter)

    # 添加两个Handler
    logger.addHandler(fh)
    # logger.addHandler(ch)


def log_training_settings(lr, num_epochs, batch_size, optim, scheduler, loss):
    # log训练信息
    logging.debug('Learning rate: '+str(lr))
    logging.debug('Max epochs: '+str(num_epochs))
    logging.debug('Batch size: '+str(batch_size))
    logging.debug('\n'+str(optim))
    logging.debug('Scheduler: '+scheduler.__module__)
    logging.debug(scheduler.state_dict())
    logging.debug(loss)
    pass
