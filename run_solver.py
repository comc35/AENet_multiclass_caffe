'''
just for the test of prorotxt
'''
import caffe

solver = caffe.SGDSolver('dummy_solver_val.prototxt')
solver.test_nets[0].share_with(solver.net)
solver.step(1)

