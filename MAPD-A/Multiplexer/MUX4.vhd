LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
ENTITY mux4 IS
    PORT (
        a1, a2, a3, a4 : IN STD_LOGIC_VECTOR(2 DOWNTO 0);
        sel
        : IN STD_LOGIC_VECTOR(1 DOWNTO 0);
        b
        : OUT STD_LOGIC_VECTOR(2 DOWNTO 0));
END mux4;
ARCHITECTURE rtl OF mux4 IS
    COMPONENT mux2 IS
        PORT (
            a1, a2 : IN STD_LOGIC_VECTOR(2 DOWNTO 0);
            sel
            : IN STD_LOGIC;
            b
            : OUT STD_LOGIC_VECTOR(2 DOWNTO 0));
    END COMPONENT mux2;
    SIGNAL muxA_out, muxB_out : STD_LOGIC_VECTOR(2 DOWNTO 0);
BEGIN
    muxA : mux2
    PORT MAP(
        a1 => a1, a2 => a2,
        sel => sel(0),
        b
        => muxA_out);
    muxB : mux2
    PORT MAP(
        a1 => a3, a2 => a4,
        sel => sel(0),
        b
        => muxB_out);
    muxOut : mux2
    PORT MAP(
        a1 => muxA_out, a2 => muxB_out,
        sel => sel(1),
        b
        => b);
END rtl;