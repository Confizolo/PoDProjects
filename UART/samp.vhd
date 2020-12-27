LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;
ENTITY samp IS
    PORT (
        CLK, UART_RX : IN STD_LOGIC;
        BAUD_OUT : OUT STD_LOGIC
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
            clk, uart_in, pulse_out : IN STD_LOGIC;
            enable : OUT STD_LOGIC
        );
    END COMPONENT samp_sm;
    COMPONENT pulse_gen IS
        PORT (
            clk, enable : IN STD_LOGIC;
            samp_out : OUT STD_LOGIC
        );
    END COMPONENT pulse_gen;
    SIGNAL ENABLE : STD_LOGIC;
    SIGNAL PULSE_OUT : STD_LOGIC;
BEGIN -- architecture rtl
    gen : pulse_gen
    PORT MAP(
        CLK => clk,
        ENABLE => enable,
        samp_out => PULSE_OUT 
    );
    sm : samp_sm
    PORT MAP(
        CLK => clk,
        uart_in => UART_RX,
        PULSE_OUT => pulse_out,
        ENABLE => enable
    );
    del : samp_del
    PORT MAP(
        CLK => clk,
        PULSE_OUT => pulse_out,
        BAUD_OUT => baud_out
    );

END ARCHITECTURE rtl;