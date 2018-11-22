#!/usr/bin/env python
# -*-coding:utf-8-*-
# @Author : Weiqun Wu
# @Time : 2018-11-23

class ConfigFactory:

    def __init__(self, name='mcnn'):
        self.name = name
        self.batch_size = 1
        self.lr = 1e-6
        self.lr_decay = 0.9
        self.momentum = 0.9
        self.total_iters = 200000
        self.max_ckpt_keep = 2000
        self.ckpt_router = '../ckpts/' + self.name + r'/'
        self.log_router = '../logs/' + self.name + r'/'

    def display_configs(self):
        msg = '''
        ------------ info of %s model -------------------
        batch size              : %s
        learing rate            : %f
        learing rate decay      : %f
        momentum                : %f
        iter num                : %s
        max ckpt keep           : %s
        ckpt router             : %s
        log router              : %s
        ------------------------------------------------
        ''' % (self.name, self.batch_size, self.lr, self.lr_decay, self.momentum, self.total_iters, self.max_ckpt_keep,
               self.ckpt_router, self.log_router)
        print(msg)
        return msg


if __name__ == '__main__':
    configs = ConfigFactory()
    configs.display_configs()
