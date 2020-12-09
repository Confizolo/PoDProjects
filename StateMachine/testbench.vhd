LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
ENTITY testbench IS
END testbench;
ARCHITECTURE tb OF testbench IS
    COMPONENT my_fsm1 IS
        PORT (
            TOG_EN, CLK, CLR : IN STD_LOGIC;
            Z1
            : OUT STD_LOGIC);
    END COMPONENT;
    SIGNAL TOG_EN, CLK, CLR, Z1: STD_LOGIC;
BEGIN
    DUT : my_fsm1 PORT MAP(TOG_EN, CLK, CLR, Z1);
    PROCESS
    BEGIN
        TOG_EN <= '0';
        WAIT FOR 1 ns;
        CLK <= '1';
        WAIT FOR 1 ns;
        TOG_EN <= '1';
        WAIT FOR 1 ns;
        CLK <= '0';
        WAIT FOR 1 ns;
        --rst<='0';
        WAIT FOR 1 ns;
        CLK <= '1';
        WAIT FOR 1 ns;
        TOG_EN <= '1';
        WAIT FOR 1 ns;
        CLK <= '0';
        WAIT FOR 1 ns;
        TOG_EN <= '0';
        WAIT FOR 1 ns;
        CLK <= '1';
        WAIT FOR 1 ns;
        TOG_EN <= '0';
        WAIT FOR 1 ns;
        CLK <= '0';
        WAIT FOR 1 ns;
        WAIT;
    END PROCESS;
END tb;