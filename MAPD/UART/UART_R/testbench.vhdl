LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
ENTITY testbench IS
END testbench;
ARCHITECTURE tb OF testbench IS
    COMPONENT samp IS
    PORT (
        CLK, UART_RX : IN STD_LOGIC;
        BAUD_OUT : OUT STD_LOGIC
    );
    END COMPONENT;
    SIGNAL CLK : std_logic := '0';
    SIGNAL UART_RX : STD_LOGIC := '1';
    SIGNAL BAUD_OUT : std_logic := '0';
BEGIN
    DUT : samp PORT MAP(CLK,UART_RX,BAUD_OUT);
    CLK <= not CLK after 10 ns;
    PROCESS
    BEGIN
        WAIT FOR 30 ns;
        UART_RX <= '0';
        WAIT FOR 30 ns;
        UART_RX <= '1';
        WAIT;
    END PROCESS;
END tb;