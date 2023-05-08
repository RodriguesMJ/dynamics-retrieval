# Dynamics Retrieval

Dynamics retrieval (SSA, LPSA, NLSA) methods with application to time-resolved serial crystallography data and other (synthetic, climate).


Basic usage for conda users: 
(also see docs/installation.rst)

git clone git@github.com:CeciliaCasadei/dynamics-retrieval.git

Generate your project dedicated environment:
conda create --name py27 python=2.7
conda activate py27
conda list
pip install -e .

Verify that dynamics_retrieval is an installed package.
conda list

Verify that imports of dynamics_retrieval modules work:
cd workflows
python test_package.py