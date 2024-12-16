# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import time
import wandb
from datetime import datetime

# Bittensor
import bittensor as bt

# import base validator class which takes care of most of the boilerplate
from template.base.validator import BaseValidatorNeuron

# Bittensor Validator Template:
from template.protocol import Challenge
from neurons.validator.get_synapse import get_synapse
from neurons.validator.reward import get_rewards
from template.utils.uids import get_miner_uids


class Validator(BaseValidatorNeuron):
    """
    Your validator neuron class. You should use this class to define your validator's behavior. In particular, you should replace the forward function with your own logic.

    This class inherits from the BaseValidatorNeuron class, which in turn inherits from BaseNeuron. The BaseNeuron class takes care of routine tasks such as setting up wallet, subtensor, metagraph, logging directory, parsing config, etc. You can override any of the methods in BaseNeuron if you need to customize the behavior.

    This class provides reasonable default behavior for a validator such as keeping a moving average of the scores of the miners and using them to set weights at the end of each epoch. Additionally, the scores are reset for new hotkeys at the end of each epoch.
    """

    def __init__(self, config=None):
        super(Validator, self).__init__(config=config)

        bt.logging.info("load_state()")
        self.load_state()
        self.init_wandb()
        
    def __del__(self):
        if self.wandb_run is not None:
            self.wandb_run.finish()
    
    def init_wandb(self):
        self.wandb_run = None
        self.wandb_run_start = None
        
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is not None:
            bt.logging.info("Logging into wandb.")
            wandb.login(key=wandb_api_key)
        else:
            bt.logging.warning("WANDB_API_KEY not found in environment variables.")
            return
        
        if not self.config.wandb.off:
            if self.config.subtensor.network == "finney":
                self.wandb_project_name = "legaltensor"
            else:
                self.wandb_project_name = "legaltensor-test"
            self.wandb_entity = "legaltensor"
            self.new_wandb_run()
    
    def new_wandb_run(self):
        """Creates a new wandb run to save information to."""
        now = datetime.now()
        self.wandb_run_start = now
        run_id = now.strftime("%Y-%m-%d-%H-%M-%S")
        name = f"validator-{self.uid}"
        self.wandb_run = wandb.init(
            project=self.wandb_project_name,
            name=name,
            entity=self.wandb_entity,
            config={
                "uid": self.uid,
                "run_id": run_id,
                "hotkey": self.wallet.hotkey.ss58_address,
                "type": "validator",
            },
            reinit=True,
        )
        bt.logging.debug(f"Started a new wandb run: {name}")

    async def forward(self):
        """
        The forward function is called by the validator every time step.

        It is responsible for querying the network and scoring the responses.

        Args:
            self (:obj:`bittensor.neuron.Neuron`): The neuron object which contains all the necessary state for the validator.

        """
        # TODO(developer): Define how the validator selects a miner to query, how often, etc.
        # get_random_uids is an example method, but you can replace it with your own.
        miner_uids = get_miner_uids(self, k=self.config.neuron.sample_size)
        # miner_uids = [1]
        miner_axons = [self.metagraph.axons[uid] for uid in miner_uids]
        bt.logging.debug(f'Available miner_uids: {miner_uids}')
        bt.logging.debug(f'Available miner_axons: {miner_axons}')

        synapse, answer = get_synapse()
        bt.logging.debug(f'Generate synapse: {synapse}')
        # The dendrite client queries the network.
        responses = await self.dendrite(
            axons=miner_axons,
            synapse=synapse,
            deserialize=True,
            timeout=60
        )

        # Log the results for monitoring purposes.
        bt.logging.info(f"Received responses: {responses}")

        # TODO(developer): Define how the validator scores responses.
        # Adjust the scores based on responses from miners.
        rewards = get_rewards(self, task_type=synapse.task_type, answer=answer, responses=responses)

        bt.logging.info(f"Scored responses: {rewards}")
        # Update the scores based on the rewards. You may want to define your own update_scores function for custom behavior.
        self.update_scores(rewards, miner_uids)
        time.sleep(5)


# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    with Validator() as validator:
        while True:
            bt.logging.info(f"Validator running... {time.time()}")
            time.sleep(5)
