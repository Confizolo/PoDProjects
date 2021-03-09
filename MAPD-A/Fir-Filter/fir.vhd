LIBRARY ieee;
USE ieee.std_logic_1164.ALL;
USE ieee.numeric_std.ALL;

ENTITY fir_filter IS
    PORT (
        i_clk : IN STD_LOGIC;
        i_rstb : IN STD_LOGIC;
        -- data input
        i_valid : IN STD_LOGIC;
        i_data : IN STD_LOGIC_VECTOR(7 DOWNTO 0);
        -- filtered data 
        o_valid : OUT STD_LOGIC;
        o_data : OUT STD_LOGIC_VECTOR(7 DOWNTO 0));
END fir_filter;

ARCHITECTURE rtl OF fir_filter IS
BEGIN

    --    p_input : PROCESS (i_rstb, i_clk)
    --    BEGIN
    --        IF (i_rstb = '0') THEN
    --            v_coeff <= (OTHERS => (OTHERS => '0'));
    --        ELSIF (rising_edge(i_clk)) THEN
    --            v_coeff(0) <= signed(coeff_0);
    --            v_coeff(1) <= signed(coeff_1);
    --            v_coeff(2) <= signed(coeff_2);
    --            v_coeff(3) <= signed(coeff_3);
    --        END IF;
    --    END PROCESS p_input;
    processing : PROCESS (i_rstb, i_clk) IS
        TYPE t_coeff IS ARRAY (0 TO 3) OF signed(7 DOWNTO 0);
        TYPE t_mult IS ARRAY (0 TO 3) OF signed(15 DOWNTO 0);

        -- coefficient
        VARIABLE coeff_0 : STD_LOGIC_VECTOR(7 DOWNTO 0) := X"B2";
        VARIABLE coeff_1 : STD_LOGIC_VECTOR(7 DOWNTO 0) := X"01";
        VARIABLE coeff_2 : STD_LOGIC_VECTOR(7 DOWNTO 0) := X"ff";
        VARIABLE coeff_3 : STD_LOGIC_VECTOR(7 DOWNTO 0) := X"ff";
        VARIABLE v_coeff : t_coeff := (signed(coeff_0), signed(coeff_1), signed(coeff_2), signed(coeff_3));

        VARIABLE v_mult : t_mult := (OTHERS => (OTHERS => '0'));
        VARIABLE v_add_0 : signed(16 DOWNTO 0);
        VARIABLE v_add_1 : signed(16 DOWNTO 0);

        VARIABLE result : signed(15 + 2 DOWNTO 0);
    BEGIN
        IF (i_rstb = '0') THEN
            v_mult := (OTHERS => (OTHERS => '0'));
            result := (OTHERS => '0');
            o_data <= (OTHERS => '0');

        ELSIF (rising_edge(i_clk)) AND (i_valid = '1') THEN
            FOR k IN 0 TO 3 LOOP
                v_mult(k) := v_coeff(k) * signed(i_data);
            END LOOP;
            v_add_0 := resize(v_mult(0), 17) + resize(v_mult(1), 17);
            v_add_1 := resize(v_mult(2), 17) + resize(v_mult(3), 17);
            result := resize(v_add_0, 18) + resize(v_add_1, 18);
            o_valid <= '1';
            o_data <= STD_LOGIC_VECTOR(result(17 DOWNTO 10));
        ELSE
            o_valid <= '1';
        END IF;
    END PROCESS processing;
END rtl;