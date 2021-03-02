LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;
ENTITY uart IS
    PORT (
        txclk, valid : IN STD_LOGIC;
        txDATA : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        UARTout, txBUSY : OUT STD_LOGIC
    );
END uart;
ARCHITECTURE rtl OF uart IS
    COMPONENT generator IS
        PORT (
            clk : IN STD_LOGIC;
            y : OUT STD_LOGIC
        );
    END COMPONENT generator;
    COMPONENT my_fsm IS
        PORT (
            DATA_EN : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
            CLK, BOUD_OUT, VALID : IN STD_LOGIC;
            UARTtx, BUSY
            : OUT STD_LOGIC);
    END COMPONENT my_fsm;
    SIGNAL boudclk : STD_LOGIC;
BEGIN -- architecture rtl
    gen : generator
    PORT MAP(
        clk => txclk,
        y => boudclk
    );
    sm : my_fsm
    PORT MAP(
        DATA_EN => txDATA,
        valid => VALID,
        BOUD_OUT => boudclk,
        CLK => txclk,
        UARTtx => UARTout,
        BUSY => txBUSY
    );
END ARCHITECTURE rtl;