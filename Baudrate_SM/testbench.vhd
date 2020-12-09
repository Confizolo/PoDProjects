LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
ENTITY testbench IS
END testbench;
ARCHITECTURE tb OF testbench IS
    COMPONENT uart IS
        PORT (
            txclk, valid : IN STD_LOGIC;
            txDATA : IN STD_LOGIC_VECTOR(0 TO 7);
            UARTout, txBUSY : OUT STD_LOGIC
        );
    END COMPONENT;
    SIGNAL txclk : STD_LOGIC := '0';
    SIGNAL valid : STD_LOGIC;
    SIGNAL txDATA : STD_LOGIC_VECTOR(0 TO 7);
    SIGNAL UARTout, txBUSY : STD_LOGIC := '0';
BEGIN
    DUT : uart PORT MAP(txclk, valid, txDATA, UARTout, txBUSY);
    txclk <= NOT txclk AFTER 10 ns;
    PROCESS
    BEGIN
        WAIT FOR 1000 ns;
        valid <= '1';
        txDATA <= "10101010";
        
        WAIT ON txBUSY;
            valid <= '0';
        WAIT;
    END PROCESS;
END tb;