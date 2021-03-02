LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
ENTITY testbench IS
END testbench;
ARCHITECTURE tb OF testbench IS
    COMPONENT uart IS
        PORT (
            txclk, valid : IN STD_LOGIC;
            txDATA : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
            UARTout, txBUSY : OUT STD_LOGIC
        );
    END COMPONENT;
    SIGNAL txclk : STD_LOGIC := '0';
    SIGNAL valid : STD_LOGIC;
    SIGNAL txDATA : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL UARTout, txBUSY : STD_LOGIC := '0';
BEGIN
    DUT : uart PORT MAP(txclk, valid, txDATA, UARTout, txBUSY);
    txclk <= NOT txclk AFTER 10 ns;
    PROCESS
    BEGIN
        WAIT FOR 1000 ns;
        valid <= '1';
        txDATA <= "10101010";
        
        WAIT UNTIL txBUSY='1';
        valid<='0';
        WAIT UNTIL txBUSY='0';
        valid <= '1';
        txDATA <= "11111010";
        WAIT UNTIL txBUSY='0';
        valid<='0';
        WAIT;
    END PROCESS;
END tb;