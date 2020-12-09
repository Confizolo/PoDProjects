LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
ENTITY testbench IS
END testbench;
ARCHITECTURE tb OF testbench IS
    COMPONENT dff IS
        PORT (
            clk : IN STD_LOGIC := '0';
            rst : IN STD_LOGIC;
            t
            : IN STD_LOGIC := '0';
            q
            : INOUT STD_LOGIC;
            d: INOUT STD_LOGIC := '0' 
            );
    END COMPONENT;
    SIGNAL clk, rst, t,q,d : STD_LOGIC;
BEGIN
    DUT : dff PORT MAP(clk, rst, t,q,d);
    PROCESS
    BEGIN
        t <= '0';
        WAIT FOR 1 ns;
        clk <= '1';
        WAIT FOR 1 ns;
        t <= '1';
        WAIT FOR 1 ns;
        clk <= '0';
        WAIT FOR 1 ns;
        --rst<='0';
        WAIT FOR 1 ns;
        clk <= '1';
        WAIT FOR 1 ns;
        t <= '1';
        WAIT FOR 1 ns;
        clk <= '0';
        WAIT FOR 1 ns;
        t <= '0';
        WAIT FOR 1 ns;
        clk <= '1';
        WAIT FOR 1 ns;
        t <= '0';
        WAIT FOR 1 ns;
        clk <= '0';
        WAIT FOR 1 ns;
        WAIT;
    END PROCESS;
END tb;