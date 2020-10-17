# DaisyGANv6
This fork aims to take DaisyGANv5 into a multi-process model of workload distribution.

The concept here is to have a seperate network per word length sentence; e.g, 1 - 16 word sentences have 16 different networks.

This allows muli-process parallelisation over multiple CPU cores and threads, data can be aggregated from the DaisyGAN outputs however desired by scripts in higher level languages. I have provided a telegram bot aggregator in PHP `tgbot.php`.

- **There is only one command line of arguments for DaisyGANv6 and it is:**<br>
`./cfdgan <number of words per sentence to digest> <first layer size> <hidden layer size> <digest lines amount> <output lines amount> <minimum output lines per second (MOLPS) before recompute> <timeout in seconds to find weights with a fail variance of 70 or more, after timeout, settle for next best lowest number> <update tick in seconds for the service to check and digest the streams>`
