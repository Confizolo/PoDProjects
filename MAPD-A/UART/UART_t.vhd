LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;
ENTITY uart_t IS
    PORT (
        txclk, valid : IN STD_LOGIC;
        tx_data : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        uart_out, tx_busy : OUT STD_LOGIC
    );
END uart_t;
ARCHITECTURE rtl OF uart_t IS
    COMPONENT generator_1 IS
        PORT (
            clk : IN STD_LOGIC;
            baud_out : OUT STD_LOGIC
        );
    END COMPONENT generator_1;
    COMPONENT fsm IS
        PORT (
            data_en : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
            clk, baud_out, valid : IN STD_LOGIC;
            uart_tx, busy : OUT STD_LOGIC);
    END COMPONENT fsm;
    SIGNAL baudclk : STD_LOGIC;
BEGIN -- architecture rtl
    gen : generator_1
    PORT MAP(
        clk => txclk,
        baud_out => baudclk
    );
    sm : fsm
    PORT MAP(
        data_en => tx_data,
        valid => valid,
        baud_out => baudclk,
        clk => txclk,
        uart_tx => uart_out,
        busy => tx_busy
    );
END ARCHITECTURE rtl;