LIBRARY IEEE;
USE IEEE.std_logic_1164.ALL;
ENTITY total IS
    PORT (
        clk, valid : IN STD_LOGIC;
        txDATA : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        UARTout, txBUSY : OUT STD_LOGIC
    );
END total;
ARCHITECTURE rtl OF total IS
    COMPONENT UART_R IS
        PORT (
            CLK, UART_RX : IN STD_LOGIC;
            VALID, DATA : OUT STD_LOGIC
        );
    END COMPONENT;
    COMPONENT uart IS
        PORT (
            txclk, valid : IN STD_LOGIC;
            txDATA : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
            UARTout, txBUSY : OUT STD_LOGIC
        );
    END COMPONENT;
    SIGNAL uart_rx : STD_LOGIC;
BEGIN
    gen : uart
    PORT MAP(
        txclk => clk,
        valid => valid,
        txDATA => txDATA,
        UARTout => uart_rx,
        txBUSY => txBUSY
    );
    receiv : UART_R
    PORT MAP(
        clk => CLK,
        uart_rx => UART_RX,
        DATA => UARTout
    );
END ARCHITECTURE rtl;