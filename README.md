# DaisyGANv6
This fork aims to take DaisyGANv5 into a multi-process model of workload distribution.

The concept here is to have a separate network per word length sentence; e.g, 1 - 16 word sentences have 16 different networks.

This allows multi-process parallelisation over multiple CPU cores and threads, data can be aggregated from the DaisyGAN outputs however desired by scripts in higher-level languages. I have provided a telegram bot aggregator in PHP `tgbot.php`.

As DaisyGAN expands over multiple processes the challenge becomes more a problem of aggregating enough input data, having a bot added to popular groups is no easy task, I suggest to aggregate the number of messages required to train DaisyGANv6 in an adequate time period that one sets up a regular account connected to lots of popular telegram groups and additionally uses a [MTProto](https://www.google.com/search?&q=telegram+proto+libraries) library to aggregate the messages from the regular telegram account.
<br>
- **There is only one command line of arguments for DaisyGANv6 and it is:**<br>
`./cfdgan <number of words per sentence to digest> <first layer size> <hidden layer size> <digest lines amount> <output lines amount> <minimum output lines per second (MOLPS) before recompute> <timeout in seconds to find weights with a fail variance of 70 or more, after timeout, settle for next best lowest number> <update tick in seconds for the service to check and digest the input stream>`
