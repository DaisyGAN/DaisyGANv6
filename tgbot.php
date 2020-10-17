<?php

    $token = '<BOT KEY>';
    $j = json_decode(file_get_contents("php://input"));

    function appendFileUnique($fp, $line)
    {
        $data = file_get_contents($fp);
        if(strpos($data, $line . "\n") === false)
            file_put_contents($fp, $line . "\n", FILE_APPEND | LOCK_EX);
    }

    function randomQuote()
    {
        $f = "port/" . rand(2, 15) . "/out.txt";
        $st = time();
        while(file_exists($f) == FALSE || filesize($f) == 0)
        {
            $f = "port/" . rand(1, 16) . "/out.txt";
            if(time() - $st > 3)
                return "No data.";
        }
        $file = file($f); 
        return $file[rand(0, count($file) - 1)];
    }

    function garbageQuote()
    {
        $f = "port/1/out.txt";
        if(rand(0, 100) <= 50)
            $f = "port/16/out.txt";
        $st = time();
        while(file_exists($f) == FALSE || filesize($f) == 0)
        {
            if($f == "port/16/out.txt")
                $f = "port/1/out.txt";
            else
                $f = "port/16/out.txt";
            if(time() - $st > 3)
                return "No data.";
        }
        $file = file($f); 
        return $file[rand(0, count($file) - 1)];
    }

    if(isset($j->{'message'}->{'text'}) && isset($j->{'message'}->{'chat'}->{'id'}))
    {
        if(strpos($j->{'message'}->{'text'}, "/info") !== FALSE)
        {
            $s0 = 0;
            $s1 = 0;
            $s2 = 0;
            $s3 = 0;
            $s4 = 0;
            $s5 = 0;
            $s6 = 0;
            $s7 = 0;
            $s8 = 0;
            $loaded = 0;
            $stat = "";
            for($i = 1; $i <= 16; $i++)
            {
                $f = "port/" . $i . "/stat.txt";
                if(file_exists($f))
                {
                    $file = file($f); 
                    $s0 += $file[0];
                    $s1 += $file[1];
                    $s2 += $file[2];
                    $s3 += $file[3];
                    $s4 += $file[4];
                    $s5 += $file[5];
                    $s6 += $file[6];
                    $s7 += $file[7];
                    $s8 += $file[8];
                    $stat .= "[".$i."] Core\nRMSE: " . $file[0] . "Variance: " . $file[1] . "TTMin: " . $file[2] . "L-Rate: " . $file[3] . "Dropout: " . $file[4] . "Momentum: " . $file[5] . "Alpha: " . $file[6] . "Neurons: " . number_format($file[7]) . "\nWeights: " . number_format($file[8]) . "\n";
                    if(isset($file[9]))
                        $stat .= $file[9];
                    $stat .= "\n";
                    $loaded++;
                }
            }
            $s0 /= $loaded;
            $s1 /= $loaded;
            $s2 /= $loaded;
            $s3 /= $loaded;
            $s4 /= $loaded;
            $s5 /= $loaded;
            $s6 /= $loaded;

            $stat .= "[Totals]\nAvg RMSE: " . number_format($s0, 6) . "\nAvg Variance: " . number_format($s1, 1) . "\nAvg TTMin: " . number_format($s2, 2) . "\nAvg L-Rate: " . number_format($s3, 6) . "\nAvg Dropout: " . number_format($s4, 6) . "\nAvg Momentum: " . number_format($s5, 6) . "\nAvg Alpha: " . number_format($s6, 6) . "\nNeurons: " . number_format($s7) . "\nWeights: " . number_format($s8) . "\n";

            $chatid = $j->{'message'}->{'chat'}->{'id'};
            file_get_contents("https://api.telegram.org/bot" . $token . "/sendMessage?chat_id=" . $chatid . "&text=".urlencode($stat));
            http_response_code(200);
            exit;
        }

        if(strpos($j->{'message'}->{'text'}, "/quote") !== FALSE)
        {
            $chatid = $j->{'message'}->{'chat'}->{'id'};
            file_get_contents("https://api.telegram.org/bot" . $token . "/sendMessage?chat_id=" . $chatid . "&text=".urlencode(randomQuote()));
            http_response_code(200);
            exit;
        }

        if(strpos($j->{'message'}->{'text'}, "/trash") !== FALSE)
        {
            $chatid = $j->{'message'}->{'chat'}->{'id'};
            file_get_contents("https://api.telegram.org/bot" . $token . "/sendMessage?chat_id=" . $chatid . "&text=".urlencode(garbageQuote()));
            http_response_code(200);
            exit;
        }

        $pp = explode(' ', $msg);
        $pps = array_slice($pp, 0, 16);

        $str = "";
        foreach($pps as $p)
            if(strlen($p) <= 250 && $p != "" && $p != " ")
                $str .= str_replace("\n", "", $p) . " ";
        $str = rtrim($str, ' ');

        appendFileUnique("port/" . count($pps) . "/botmsg.txt", str_replace("\n", "", substr($str, 0, 4090)));

        foreach($pp as $p)
            if(strlen($p) <= 250 && $p != "" && $p != " ")
                appendFileUnique("botdict.txt", str_replace("\n", "", substr($p, 0, 250)));
    }

    http_response_code(200);

?>
