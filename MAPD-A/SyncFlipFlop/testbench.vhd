LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
ENTITY testbench IS
END testbench;
ARCHITECTURE tb OF testbench IS
    COMPONENT sdff IS
    PORT (
        clk: IN std_logic;
        rst1, rst2, d1 : IN STD_LOGIC;
        q2 : OUT STD_LOGIC
    );
    END COMPONENT;
    SIGNAL clk,rst1,rst2,d1,q2 : STD_LOGIC;
BEGIN
    DUT : sdff PORT MAP(clk,rst1,rst2,d1,q2);
    PROCESS
    BEGIN
        clk <= '0';
        d1<='0';
        WAIT FOR 1 ns;
        clk <= '1';
        WAIT FOR 2 ns;
        clk <= '0';
        WAIT FOR 1 ns;
        d1<='1';
        WAIT FOR 1 ns;
        clk <= '1';
        WAIT FOR 2 ns;
        clk <= '0';
        WAIT FOR 1 ns;
        d1<='0';
        WAIT FOR 1 ns;
        clk <= '1';
        WAIT FOR 1 ns;
        d1<= '1';
        WAIT FOR 1 ns;
        clk <= '0';
        WAIT FOR 2 ns;
        clk <= '1';
        WAIT FOR 1 ns;
        d1<='0';
        WAIT;
    END PROCESS;
END tb;