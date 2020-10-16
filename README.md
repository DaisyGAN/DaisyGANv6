# DaisyGANv6
This fork aims to take DaisyGANv5 into a multi-process model of workload distribution.

The concept here is to have a seperate network per word length sentence; e.g, 1 - 16 word sentences have 16 different networks.

This allows muli-process parallelisation over multiple CPU cores and threads, data can be aggregated from the DaisyGAN outputs however desired by scripts in higher level languages. I have provided a telegram bot aggreagtor in PHP `tgbot.php`.

- **There is only one command line argumentfor DaisyGANv6 and it is:**<br>
`./cfdgan <number of words per sentence to digest> <first layer size> <hidden layer size> <digest lines amount> <output lines amount>`
