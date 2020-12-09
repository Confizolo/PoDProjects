LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
ENTITY testbench IS
END testbench;
ARCHITECTURE tb OF testbench IS
    COMPONENT generator IS
    PORT (
        clk : IN STD_LOGIC ;
        y : OUT STD_LOGIC
    );
    END COMPONENT;
    SIGNAL clk : std_logic := '0';
    SIGnaL y : STD_LOGIC;
BEGIN
    DUT : generator PORT MAP(clk,y);
    clk <= not clk after 10 ns;
    PROCESS
    BEGIN
        WAIT;
    END PROCESS;
END tb;