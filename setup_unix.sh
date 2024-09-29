python -m pip install --user virtualenv

python -m virtualenv env

source ./env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

deactivate