LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;
ENTITY samp IS
    PORT (
        clk, uart_rx : IN STD_LOGIC;
        baud_out : OUT STD_LOGIC
    );
END samp;
ARCHITECTURE rtl OF samp IS
    COMPONENT samp_del IS
        PORT (
            clk, pulse_out : IN STD_LOGIC;
            baud_out : OUT STD_LOGIC
        );
    END COMPONENT samp_del;
    COMPONENT samp_sm IS
        PORT (
            clk, uart_rx, pulse_out : IN STD_LOGIC;
            enable : OUT STD_LOGIC
        );
    END COMPONENT samp_sm;
    COMPONENT pulse_gen IS
        PORT (
            clk, enable : IN STD_LOGIC;
            samp_out : OUT STD_LOGIC
        );
    END COMPONENT pulse_gen;
    SIGNAL enable : STD_LOGIC;
    SIGNAL pulse_out : STD_LOGIC;
BEGIN -- architecture rtl
    gen : pulse_gen
    PORT MAP(
        clk => clk,
        enable => enable,
        samp_out => pulse_out
    );
    sm : samp_sm
    PORT MAP(
        clk => clk,
        uart_rx => uart_rx,
        pulse_out => pulse_out,
        enable => enable
    );
    del : samp_del
    PORT MAP(
        clk => clk,
        pulse_out => pulse_out,
        baud_out => baud_out
    );

END ARCHITECTURE rtl;