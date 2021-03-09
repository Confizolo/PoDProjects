LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;
ENTITY uart_r IS
    PORT (
        clk, uart_rx : IN STD_LOGIC;
        valid : OUT STD_LOGIC;
        data : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
    );
END uart_r;
ARCHITECTURE rtl OF uart_r IS
    COMPONENT main_sm IS
        PORT (
            clk, uart_rx, baud_in : IN STD_LOGIC;
            valid : OUT STD_LOGIC;
            rec_out : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
        );
    END COMPONENT main_sm;
    COMPONENT samp IS
        PORT (
            clk, uart_rx : IN STD_LOGIC;
            baud_out : OUT STD_LOGIC
        );
    END COMPONENT samp;
    SIGNAL baud : STD_LOGIC;
BEGIN -- architecture rtl
    sm : main_sm
    PORT MAP(
        clk => clk,
        uart_rx => uart_rx,
        baud_in => baud,
        valid => valid,
        rec_out => data
    );
    smp : samp
    PORT MAP(
        clk => clk,
        uart_rx => uart_rx,
        baud_out => baud
    );
END ARCHITECTURE rtl;