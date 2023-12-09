#python COBA_brainpy_np.py
#
#python COBA_pynn.py neuron COBA
#
#python COBA_nest.py
#
#python COBA_ANNarchy.py 1
#python COBA_ANNarchy.py 2
#python COBA_ANNarchy.py 4
#

#python COBA_brian2.py cpp_standalone
#python COBA_brian2.py cpp_standalone 12
python COBA_brian2.py --backend cpp_standalone --threads 1 --dtype f32
python COBA_brian2.py --backend cpp_standalone --threads 12 --dtype f32
python COBA_brian2.py --backend genn --dtype f32
python COBA_brian2.py --backend cuda_standalone --dtype f32
python COBA_brainpy_jax.py 1
python COBA_brainpy_jax.py 0


#python COBA_pynn.py nest --threads 12
#python COBA_pynn.py nest --threads 1
#python COBA_pynn.py neuron --threads 1
#python COBA_pynn.py neuron --threads 12
