

class DummyTestCase(object):

    def __init__(self):
        try:
            self.setUp()
        except AttributeError:
            pass

    def assertEqual(self, first, second):
        print 'are{} equal'.format(' not' if first != second else '')
