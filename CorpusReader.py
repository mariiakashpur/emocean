import abc

class CorpusReader(object):
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def get_instances(self, folder):
        pass
