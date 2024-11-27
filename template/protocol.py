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

import typing
import bittensor as bt

# TODO(developer): Rewrite with your protocol definition.

# This is the protocol for the Challenge miner and validator.
# It is a simple request-response protocol where the validator sends a request
# to the miner, and the miner responds with a Challenge response.

# ---- miner ----
# Example usage:
#   def Challenge( synapse: Challenge ) -> Challenge:
#       synapse.Challenge_output = synapse.Challenge_input + 1
#       return synapse
#   axon = bt.axon().attach( Challenge ).serve(netuid=...).start()

# ---- validator ---
# Example usage:
#   dendrite = bt.dendrite()
#   Challenge_output = dendrite.query( Challenge( Challenge_input = 1 ) )
#   assert Challenge_output == 2


class Challenge(bt.Synapse):
    """
    A challenge request for legal advise.

    Attributes:
    - task_type: A task string describing legal sector.
    - problem: A description of certain status according to the legal sector.
    """

    # Required request input, filled by sending dendrite caller.
    task_type: str
    problem: str

    # Optional request output, filled by receiving axon.
    result: typing.Optional[str] = None

    def deserialize(self) -> int:
        """
        Deserialize the Challenge output. This method retrieves the response from
        the miner in the form of Challenge_output, deserializes it and returns it
        as the output of the dendrite.query() call.
        """
        return self.result
