#!/bin/bash
# Install CityLearn from source WITHOUT doe_xstock and openstudio
# (LSTMDynamicsBuilding doesn't need them at runtime)
set -e
if [ -d "/tmp/citylearn_src" ]; then
    rm -rf /tmp/citylearn_src
fi
git clone https://github.com/intelligent-environments-lab/CityLearn.git /tmp/citylearn_src
cd /tmp/citylearn_src
# Remove deps that require OpenStudio C++ SDK (not needed for LSTM dynamics)
sed -i '/doe_xstock/d' requirements.txt
sed -i '/openstudio/d' requirements.txt
pip install -e .
cd -
echo "CityLearn installed successfully (without doe_xstock/openstudio)"
