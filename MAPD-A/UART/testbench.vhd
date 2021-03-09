LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
ENTITY testbench IS
END testbench;
ARCHITECTURE tb OF testbench IS
    COMPONENT total IS
        PORT (
            clk, valid_in : IN STD_LOGIC;
            tx_data : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
            tx_busy : OUT STD_LOGIC;
            g_out : OUT STD_LOGIC_VECTOR(7 DOWNTO 0)
        );
    END COMPONENT;
    SIGNAL clk : STD_LOGIC := '0';
    SIGNAL valid : STD_LOGIC := '0';
    SIGNAL tx_data : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL tx_busy : STD_LOGIC := '0';
    SIGNAL g_out : STD_LOGIC_VECTOR(7 DOWNTO 0);
BEGIN
    dut : total PORT MAP(clk, valid, tx_data, tx_busy, g_out);
    clk <= NOT clk AFTER 10 ns;
    PROCESS
    BEGIN
        WAIT FOR 1000 ns;
        valid <= '1';
        tx_data <= "10101011";

        WAIT UNTIL tx_busy = '1';
        valid <= '0';
        WAIT;
    END PROCESS;
END tb;