# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import sys
from typing import TYPE_CHECKING

from ..import_utils import _LazyModule


_import_structure = {
    "format_rewards": ["think_format_reward", "box_format_reward"],
    "other_rewards": ["get_soft_overlong_punishment"],
    "accuracy_rewards": ["think_accuracy_reward", "box_accuracy_reward"],
    "reward_new": ["cross_entropy_reward", "format_reward", "length_reward"]  # 添加这一行
}


if TYPE_CHECKING:
    from .format_rewards import think_format_reward
    from .other_rewards import get_soft_overlong_punishment
    from .accuracy_rewards import think_accuracy_reward, box_accuracy_reward
    from .reward_new import cross_entropy_reward, format_reward, length_reward  # 添加这一行


else:
    sys.modules[__name__] = _LazyModule(__name__, __file__, _import_structure, module_spec=__spec__)
