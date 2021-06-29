# mcas-gmra
gmra algorithm with ibm mcas nvram technology


# Dependencies
1) pytorch
2) tqdm (python package)
3) cmake (> 3.0)
4) mcas (and pymm)

## NOTE
-mcas installs to your system python interpreter. Therefore, the rest of these dependencies must be installed to the system interpreter as well (will still work if installed to the user (--user) system environment).

-Special note about pytorch. If you do have a prebuilt binary, then you *must* have the cuda/cudnn/nccl versions that the prebuilt version is expecting. Normally this isn't a problem when using pytorch, however this will cause the libtorch cmake toolchain to error and the build to fail.

-I am currently testing whether building pytorch from source fixes this problem, so standby.


# Building
there are git submodules so be sure to run:

git submodule update --init --recursive

Then, cd to the pymm-gmra directory. The code can be build using
python3 setup.py build

and then installed using
python3 setup.py install --user

Running the install command will also build the code (if unbuilt or modified).


Finally, if you cd into the examples directory, you will see examples of running the code on mnist. I recommend trying

mnist_nopymm_cpp.py if you do *not* have pymm installed, otherwise mnist_pymm_cpp.py

The *_cpp.py scripts will use the c++ data structures instead of the pure python versions (which the other scripts will use). Warning: pure python scripts are very, very, slow.

