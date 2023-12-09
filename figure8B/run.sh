#python COBAHH_brian2.py cpp_standalone
#python COBAHH_brian2.py cpp_standalone 12
#python COBAHH_brian2.py cpp_standalone 10
#python COBAHH_brian2.py cpp_standalone 6
#python COBAHH_brian2.py cpp_standalone 12
#python COBAHH_brian2.py cpp_standalone
#python COBAHH_brainpy.py -platform cpu -x64
#python COBAHH_brainpy.py -platform cpu

#python COBAHH_brainpy.py -platform gpu -x64
#python COBAHH_brainpy.py -platform gpu


python COBAHH_brian2.py --backend cpp_standalone --threads 1 --dtype f32
python COBAHH_brian2.py --backend cpp_standalone --threads 12 --dtype f32
python COBAHH_brian2.py --backend genn --dtype f32
python COBAHH_brian2.py --backend cuda_standalone --dtype f32

#python COBAHH_pynn.py nest --threads 12
#python COBAHH_pynn.py nest --threads 1
#python COBAHH_pynn.py neuron --threads 1
#python COBAHH_pynn.py neuron --threads 12


