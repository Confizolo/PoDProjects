LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
ENTITY top IS
    PORT (
        clk : IN STD_LOGIC;
        ser_in : IN STD_LOGIC;
        ser_out : OUT STD_LOGIC
    );
END top;
ARCHITECTURE rtl OF top IS
    COMPONENT fir_filter IS
        PORT (
            i_clk : IN STD_LOGIC;
            i_rstb : IN STD_LOGIC;
            -- data input
            i_valid : IN STD_LOGIC;
            i_data : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
            -- filtered data 
            o_valid : OUT STD_LOGIC;
            o_data : OUT STD_LOGIC_VECTOR(7 DOWNTO 0));
    END COMPONENT;

    COMPONENT uart_receiver IS
        PORT (
            clock : IN STD_LOGIC;
            uart_rx : IN STD_LOGIC;
            valid : OUT STD_LOGIC;
            received_data : OUT STD_LOGIC_VECTOR(7 DOWNTO 0));
    END COMPONENT;

    COMPONENT uart_transmitter IS
        PORT (
            clock : IN STD_LOGIC;
            data_to_send : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
            data_valid : IN STD_LOGIC;
            busy : OUT STD_LOGIC;
            uart_tx : OUT STD_LOGIC);

    END COMPONENT;
    SIGNAL rx_valid : STD_LOGIC := '0';
    SIGNAL tx_valid : STD_LOGIC := '0';
    SIGNAL tx_data : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL rx_data : STD_LOGIC_VECTOR(7 DOWNTO 0);
    SIGNAL busy : STD_LOGIC := '0';
    SIGNAL rst : STD_LOGIC := '1';
BEGIN
    uart_t : uart_transmitter
    PORT MAP(
        clock => clk,
        data_to_send => tx_data,
        data_valid => tx_valid,
        busy => busy,
        uart_tx => ser_out
    );
    uart_r : uart_receiver
    PORT MAP(
        clock => clk,
        uart_rx => ser_in,
        valid => rx_valid,
        received_data => rx_data
    );
    fir : fir_filter
    PORT MAP(
        i_clk => clk,
        i_rstb => rst,
        -- data 
        i_valid => rx_valid,
        i_data => rx_data,
        -- filte
        o_valid => tx_valid,
        o_data => tx_data
    );
END ARCHITECTURE rtl;