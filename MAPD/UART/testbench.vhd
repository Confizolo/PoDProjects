LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
ENTITY testbench IS
END testbench;
ARCHITECTURE tb OF testbench IS
    COMPONENT total IS
    PORT (
        clk, valid : IN STD_LOGIC;
        txDATA : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        UARTout, txBUSY : OUT STD_LOGIC
    );
    END COMPONENT;
    SIGNAL clk, valid : std_logic := '0';
    SIGNAL txDATA : STD_LOGIC_VECTOR(7 DOWNTO 0) ;
    SIGNAL UARTout, txBUSY : std_logic := '0';
BEGIN
    DUT : total PORT MAP(clk,valid,txDATA,UARTout, txBUSY);
    clk <= NOT clk AFTER 10 ns;
    PROCESS
    BEGIN
        WAIT FOR 1000 ns;
        valid <= '1';
        txDATA <= "10101010";

        WAIT UNTIL txBUSY = '1';
        valid <= '0';
        WAIT;
    END PROCESS;
END tb;